import java.util.Scanner;
public class T02 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        double radius, length, area, volume;
        System.out.print("Enter the radius and length of a cylinder: ");
        radius = sc.nextDouble();
        length = sc.nextDouble();
        area = radius * radius * 3.14159;
        volume = area * length;
        System.out.println("The area is " + area);
        System.out.println("The volume is " + volume);
    }
}

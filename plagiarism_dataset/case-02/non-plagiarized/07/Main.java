import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        System.out.print("Enter the radius and length of a cylinder:");
        Scanner sc = new Scanner(System.in);
        double radius = sc.nextDouble();
        int length = sc.nextInt();
        double area = radius*radius*3.14159;
        double volume = length*radius;
        System.out.println("The area is "+area);
        System.out.println("The volume is "+volume);
    }
}

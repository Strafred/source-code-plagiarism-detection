import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double jejari = input.nextDouble();
        double panjang = input.nextDouble();
        double area = jejari * jejari * 3.14159;
        double volume = area * panjang;
        System.out.println("The area is " + area);
        System.out.println("The volume of the cylinder is " + volume);
        }
}
